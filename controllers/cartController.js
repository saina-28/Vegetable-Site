const cartModel = require("../models/Cart")
const pantryModel = require("../models/Pantry")
const { insertItemInfoInCartItemsHELPER } = require('../middlewares/addCheckoutANDCartData')

const getCurrentCartItems = async (req, res)=>{
    // Also make check whether ther item is avaliable
    try{
        const cartItems = req.cartItems
        res.status(201).json({cartItems})
    }
    catch(error){
        console.log(error);
        res.status(500).json({error : error})
    }
}

const getCheckoutData = async (req, res)=>{
    try{
        const checkoutData = req.checkoutData
        res.status(201).json({ checkoutData : checkoutData })
    }
    catch(error){
        console.log(error.message);
        res.status(500).json({error : error.message})
    }
}

const removeCurrentCartItem = async (req, res)=>{
    const userid = req.userid
    const { itemid } = req.body;

    // Also make check whether ther item is avaliable
    
    // find the cart if not found create one
    try{
        let result = await cartModel.findOne({"userid": userid},{"cartItems": {$elemMatch: {itemid : itemid}}})
        if(result.cartItems.length === 0){
            //item is not in cart
            response.status(400).json({message : "Item is not in Cart"})
        }
        else{
            await cartModel.updateOne({userid : userid}, {$pull : { cartItems : {itemid : itemid}}})
        }
        // result = await cartModel.findOne({"userid": userid},{"cartItems": {$elemMatch: {itemid : itemid}}})
        res.status(201).json({message : "Item Removed from Cart", result : result.cartItems[0]})
    }
    catch(error){
        console.log(error);
        res.status(500).json(error)
    }
}

const updateCartItem = async (req, res, next)=>{

    const checkQuantityZero = (data)=>{
        if(data.quantity == 0)
        {
            req.body = {
                itemid : data.itemid
            }
            console.log("Removing Item From Cart")
            removeCurrentCartItem(req, res);
            return 0
        }
        return 1
    }

    // const userid = req.cookies.userid
    const userid = req.userid
    const {itemid, quantity} = req.body;

    // Also make check whether ther item is avaliable
    

    // update the cart
    try{
        let result = await cartModel.findOne({"userid": userid},{"cartItems": {$elemMatch: {itemid : itemid}}})
        if(result.cartItems.length === 0){
        // if(req.cartItems.length === 0){

            await cartModel.updateOne({userid : userid}, {$push : { cartItems : {itemid : itemid, quantity:quantity, createdAt : new Date() }}})
        }
        else{
            await cartModel.findOneAndUpdate(
                { userid: userid, 'cartItems.itemid': itemid },
                { $inc: { 'cartItems.$.quantity': quantity } },
                { new: true })
        }
        result = await cartModel.findOne({"userid": userid},{"cartItems": {$elemMatch: {itemid : itemid}}})
        const cart =  await insertItemInfoInCartItemsHELPER(result)
        // check quantity zero -> if zero remove from database
        if(checkQuantityZero(cart[0]))
            res.status(201).json({message : "Updated the Cart", result : cart[0]})
    }
    catch(error){
        console.log(error);
        res.status(500).json(error)
    }
}

const removeCart = async (req, res)=>{
    const { userid } = req.body
    try{
        const result = await cartModel.deleteOne({ userid : userid })
        if(result.deletedCount == 0){
            res.status(500).json({ message : "User's Cart not Found!", result : result })
        }else{            
            res.status(200).json({ message : "User Cart Deleted Successfully", result : result })
        }
    }
    catch(error){
        console.log(error);
        res.status(500).json(error)
    }
}

module.exports = { getCurrentCartItems, getCheckoutData, removeCurrentCartItem, updateCartItem, removeCart }
