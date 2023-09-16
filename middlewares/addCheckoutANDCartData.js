const cartModel = require("../models/Cart")
const pantryModel = require("../models/Pantry")


/* -------------------------------------------------------------------------- */
/*                              HELPER FUNCTIONS                              */
/* -------------------------------------------------------------------------- */

const insertItemInfoInCartItemsHELPER = async (resl)=>{
    const len = resl.cartItems.length
    let data = []
    for (let i = 0; i < len; i++)
    {
        let copy = JSON.parse(JSON.stringify(resl.cartItems[i]))
        const pantryItem = await pantryModel.findOne({_id : copy.itemid})
        if(pantryItem)
        {
            copy.item_img = pantryItem.img
            copy.item_name = pantryItem.item_name
            copy.item_price = pantryItem.item_price
            data.push(copy);
        }
        else{
            throw new Error(`Item ID : ${copy.itemid} not found in Pantry`)
        }
    }
    return data
}

/* -------------------------------------------------------------------------- */
/*                                     END                                    */
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
/*                             MAIN DRIVER FUNCION                            */
/* -------------------------------------------------------------------------- */

const addCheckoutANDCartData = async (req, res, next)=>{
    const getData = (cart)=>{
        let data = {
            totalCartValue : 0,
            gst : 0,
            deliveryCharge: 0,
            grandTotal : 0
        }
    
        cart.forEach(e => {
            data.totalCartValue += e.item_price*e.quantity
        });
        data.gst = 0.18*data.totalCartValue;
        data.deliveryCharge = 0.05*data.totalCartValue;
        data.grandTotal = data.gst+data.deliveryCharge+data.totalCartValue;
        return data;
    }
    const userid = req.cookies.userid;
    try{
        const result = await cartModel.findOne({"userid": userid})
        if(result === null || result.cartItems.length <= 0) throw new Error("Cart is Empty!")
        const cart =  await insertItemInfoInCartItemsHELPER(result)
        const checkoutData = getData(cart)

        req.cartItems = cart
        req.checkoutData = checkoutData;
        next()
    }
    catch(error){
        console.log(error.message);
        res.status(500).json({message : error.message})
    }
}

module.exports = { addCheckoutANDCartData, insertItemInfoInCartItemsHELPER }