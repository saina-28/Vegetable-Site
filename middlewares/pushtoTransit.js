const orderProcessingModel = require("../models/OrderProcessing")

const cartItemsHELPER = (cartItems)=>{
    let data = []
    for (let i = 0; i < cartItems.length; i++)
    {
        let copy = {}
        copy.itemid = cartItems[i].itemid
        copy.item_name = cartItems[i].item_name
        copy.quantity = cartItems[i].quantity
        copy.item_price = cartItems[i].item_price

        data.push(copy);
    }
    return data
}


const pushtoTransit = async (req, res, next)=>{
    const userid = req.userid
    const stripeCustomer = req.stripe_customer
    const cartItems = req.cartItems
    
    try{
        const order = await orderProcessingModel.findOne({userid : userid})
        if(order && order.stripe_session_object != {}){
            console.log(order)
            res.status(201).json({url : order.stripe_session_object.url, session : order.stripe_session_object})
            // throw new Error("Another Order Already in Processing!") 
        }
        else{
            orderProcessingModel.create({
                userid : userid,
                stripe_customer : stripeCustomer,
                cartItems : cartItemsHELPER(cartItems)
            })
            
            res.finalOrder = await orderProcessingModel.findOne({userid : userid})
            // res.status(200).json({message : "Order Successfully Moved to Transit for Payment"})
            next()
        }
    }
    catch(error){
        // next(error)
        res.status(400).json({error : error.message})
    }
}

module.exports = {pushtoTransit}